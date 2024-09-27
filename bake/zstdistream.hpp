#ifndef ZSTDISTREAM_HPP_AEBA1092_C378_4759_AFF4_B32E31EF431B
#define ZSTDISTREAM_HPP_AEBA1092_C378_4759_AFF4_B32E31EF431B

#include <memory>
#include <istream>

// Rapidobj fortunately allows us to feed it a custom data stream. Somewhat
// unfortunately, the interface for this is a std::istream.
//
// Hence, to to decompress stuff on the fly, we have to write a std::istream /
// std::streambuf adaptor. :-(

class ZStdIStream : public std::istream
{
	public:
		explicit ZStdIStream( char const* aPath );

	private:
		std::unique_ptr<std::streambuf> mInternal;
};

#endif // ZSTDISTREAM_HPP_AEBA1092_C378_4759_AFF4_B32E31EF431B
